import torch
from typing import Optional


def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModelLade():
    def __init__(self, model : torch.nn.Module, window_size = 60, guess_set_size = 60, lookahead_level = 8, topk=3) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self.window_size = window_size
        self.guess_set_size = guess_set_size
        self.lookahead_level = lookahead_level
        self.topk = topk

        self.ctx = None

    @torch.no_grad()
    def generate(self, input : torch.Tensor, ngram_cache, gamma : int) -> torch.Tensor:
        output = self._model.lade_generate(inputs=input, max_new_tokens=gamma, continue_ctx=self.ctx, continue_flag=(self.ctx != None), do_sample=False, window_size = self.window_size, guess_set_size = self.guess_set_size, lookahead_level = self.lookahead_level, ngram_cache = ngram_cache)
        self.ctx = self._model.ctx

        lst_token = int(output[0, -1])
        out_len = output.shape[-1]
        tups = []
        if self.ctx['ngram_cache'].has(lst_token):
            # tup = self.ctx['ngram_cache'].get_guess_tokens(lst_token)[-1] # get the newest ngram
            # output = torch.cat([output, torch.tensor([tup], device=output.device)], dim=-1)
            tups = self.ctx['ngram_cache'].get_guess_tokens(lst_token)[-self.topk:] # TODO

        return output, out_len, tups
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        for i in range(32):
            k = self.ctx['past_key_values'][i][0][:,:,:end_pos,:]
            v = self.ctx['past_key_values'][i][1][:,:,:end_pos,:]
            self.ctx['past_key_values'][i] = (k, v)
    
    def update_ngram_cache(self, remaining_approx_tok: torch.tensor, remaining_target_tok: torch.tensor):
        assert remaining_approx_tok.shape[-1] == remaining_target_tok.shape[-1]
        self.ctx['ngram_cache'].add_extra_ngram(remaining_approx_tok, remaining_target_tok)
    
    def update_in_place(self, key_token: int, modified_ngrams: list):
        # modified_ngrams should be like: [[1,2,3,4,5,6], [1,3,4,5,6,7]] and should not be empty
        self.ctx['ngram_cache'].update_in_place(key_token, modified_ngrams)

        


class KVCacheModelSimpleWithGuess():
    def __init__(self, model : torch.nn.Module, lookahead_level=8, debug=False) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self.debug = debug
        self.idx = 0
        self.guess_size = lookahead_level - 1

    @torch.no_grad()
    def _forward_with_kvcache(self, input_ids : torch.Tensor, guess, use_debug = False) -> torch.Tensor:
        input_len = input_ids.shape[-1]
        position_ids = torch.cat([torch.arange(self.idx, input_len, device=input_ids.device)] + [torch.arange(input_len, input_len + self.guess_size, device=input_ids.device)] * len(guess)).unsqueeze(0)     
        if self._past_key_values is None:
            guess = [[item for sublist in guess for item in sublist]]
            input_ids = torch.cat([input_ids, torch.tensor(guess, device=input_ids.device, dtype=torch.int)], dim=-1)
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits

            
            outputs = self._model.forward_with_guess(input_ids, guess_len = len(guess[0]), guess_size=self.guess_size, position_ids=position_ids)
            self._prob_history = outputs.logits
            self._past_key_values = outputs.past_key_values

            if self.debug:
                from IPython import embed; embed()
        else:
            # return the last token's logits
            guess = [[item for sublist in guess for item in sublist]]
            input_ids = torch.cat([input_ids, torch.tensor(guess, device=input_ids.device, dtype=torch.int)], dim=-1)
            cached_len = 0
            if self._past_key_values:
                cached_len = self._past_key_values[0][0].shape[2]

                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            outputs = self._model.forward_with_guess(last_input_id, past_key_values=self._past_key_values, use_cache=True, guess_len = len(guess[0]), guess_size=self.guess_size, position_ids=position_ids)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                  
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            self._past_key_values = outputs.past_key_values
        
        return 
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        for kv in self._past_key_values:
            k, v = kv
            cached_len = k.shape[2]
        self._prob_history = self._prob_history[:, :end_pos, :]
        self.idx = end_pos
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
        self.idx = end_pos
    
    @torch.no_grad()
    def confirm(self, cur_len, start_pos, length):
        assert self._past_key_values
        past_key_values_trimmed = []
        for kv in self._past_key_values:
            k, v = kv
            k[:, :, cur_len:cur_len + length, :] = k[:, :, start_pos: start_pos + length, :]
            k = k[:, :, :cur_len + length, :]
            v[:, :, cur_len:cur_len + length, :] = v[:, :, start_pos: start_pos + length, :]
            v = v[:, :, :cur_len + length, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        self._past_key_values = past_key_values_trimmed
        self._prob_history[:, cur_len:cur_len + length, :] = self._prob_history[:, start_pos: start_pos + length, :]
        self._prob_history = self._prob_history[:, :cur_len + length, :]
        self.idx = cur_len + length


class KVCacheModelSimple():
    def __init__(self, model : torch.nn.Module) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0) 
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = q.argmax(dim=-1, keepdim=True)
            x = torch.cat((x, next_tok), dim=1)
        # from IPython import embed; embed()
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
