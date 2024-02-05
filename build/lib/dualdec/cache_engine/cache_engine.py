import torch

class CacheEngine():
    def __init__(self, LEVEL, GUESS_SET_SIZE):
        self.token_map = {}
        self.level = LEVEL
        self.guess_set_size = GUESS_SET_SIZE

    
    def has(self, lst_token):
        return lst_token in self.token_map

    
    def get_guess_tokens(self, lst_token):
        return self.token_map[lst_token]


    def update(self, lst_token, max_guess):
        return

    def update_in_place(self, lst_token, n_ngrams):
        ngrams_num = len(n_ngrams)
        self.token_map[lst_token][-ngrams_num:] = n_ngrams
        return

    
    def add_extra_ngram(self, remaining_approx_tok: torch.tensor, remaining_target_tok: torch.tensor):
        tok_len = remaining_approx_tok.shape[-1]
        remaining_approx_tok = remaining_approx_tok.squeeze(0)
        remaining_target_tok = remaining_target_tok.squeeze(0)

        # kk = []

        for i in range(tok_len):
            key_tok = int(remaining_approx_tok[i])
            completion = []

            ub = min(self.level - 1, tok_len - i)

            for j in range(ub):
                completion.append(int(remaining_target_tok[i + j]))
                
                if i + j + 1 < tok_len and remaining_approx_tok[i + j + 1].to(remaining_target_tok[i + j].device) != remaining_target_tok[i + j]:
                    break
            cur_com_len = len(completion)
            if cur_com_len < self.level - 1:
                continue


            if key_tok not in self.token_map:
                self.token_map[key_tok] = []
            tup = tuple(completion)
            # from IPython import embed; embed()
            if tup in self.token_map[key_tok]: # refresh tup, making it the newset entry
                self.token_map[key_tok].remove(tup)
                self.token_map[key_tok].append(tup)
            elif len(self.token_map[key_tok]) < self.guess_set_size:
                self.token_map[key_tok].append(tup) 
            else:
                assert len(self.token_map[key_tok]) == self.guess_set_size
                self.token_map[key_tok] = self.token_map[key_tok][1:] + [tup] # sliding window: always desert the oldest one (self.token_map[lst_token][0])

        return
    
    
    
    def insert(self, lst_token, new_results, past_tokens, GUESS_SET_SIZE, LEVEL, WINDOW_SIZE):
        if GUESS_SET_SIZE != -1:
            if lst_token not in self.token_map:
                self.token_map[lst_token] = []
            tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)

            if tup in self.token_map[lst_token]:
                self.token_map[lst_token].remove(tup)
                self.token_map[lst_token].append(tup)
            elif len(self.token_map[lst_token]) < GUESS_SET_SIZE:
                self.token_map[lst_token].append(tup) 
            else:
                assert len(self.token_map[lst_token]) == GUESS_SET_SIZE
                self.token_map[lst_token] = self.token_map[lst_token][1:] + [tup] # sliding window: always desert the oldest one (self.token_map[lst_token][0])

            for i in range(1, WINDOW_SIZE):
                if past_tokens[0][i - 1] not in self.token_map:
                    self.token_map[past_tokens[0][i - 1]] = []
                tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)

                if tup in self.token_map[past_tokens[0][i - 1]]:
                    self.token_map[past_tokens[0][i - 1]].remove(tup)
                    self.token_map[past_tokens[0][i - 1]].append(tup)
                elif len(self.token_map[past_tokens[0][i - 1]]) < GUESS_SET_SIZE:
                    self.token_map[past_tokens[0][i - 1]].append(tup) 
                else:
                    assert len(self.token_map[past_tokens[0][i - 1]]) == GUESS_SET_SIZE
                    self.token_map[past_tokens[0][i - 1]] = self.token_map[past_tokens[0][i - 1]][1:] + [tup]

        else:
            if lst_token not in self.token_map:
                self.token_map[lst_token] = set()
            tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
            self.token_map[lst_token].add(tup) #add((past_tokens[1][0], new_results[0]))

            for i in range(1, WINDOW_SIZE):
                if past_tokens[0][i - 1] not in self.token_map:
                    self.token_map[past_tokens[0][i - 1]] = set()
                tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)
                self.token_map[past_tokens[0][i - 1]].add(tup) #((past_tokens[1][i], new_results[i]))