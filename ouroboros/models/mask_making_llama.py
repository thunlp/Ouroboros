import torch


def j_make_causal_mask_with_guess(
    guess_len : int, guess_size: int, input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    if guess_len > 0:
        mask[:,0] = 0
        lguess = guess_len
        if guess_size == 2:
            small_m = torch.tensor([0, torch.finfo(dtype).min]).repeat(lguess // 2)[:-1]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m, -1)
        elif guess_size == 3:
            small_m1 = torch.tensor([0, 0, torch.finfo(dtype).min]).repeat(lguess // 3)[:-1]
            small_m2 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 3)[:-2]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2)
        elif guess_size == 4:
            small_m1 = torch.tensor([0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 4)[:-1]
            small_m2 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-2]
            small_m3 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-3]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3)
        elif guess_size == 5:
            small_m1 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 5)[:-1]
            small_m2 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-2]
            small_m3 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-3]
            small_m4 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-4]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4)
        elif guess_size == 6:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 6)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-2]
            small_m3 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-3]
            small_m4 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-4]
            small_m5 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-5]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5)
        elif guess_size == 7:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 7)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-2]
            small_m3 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-3]
            small_m4 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-4]
            small_m5 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-5]
            small_m6 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-6]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5).diagonal_scatter(small_m6, -6)

        else:
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0)
            for i in range(guess_size - 1): #7 : 0 - 5
                small_l = [0] * (guess_size - i - 1)  + [torch.finfo(dtype).min] * (i + 1)
                small_m = torch.tensor(small_l).repeat(lguess // guess_size)[:-1 - i]
                mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].diagonal_scatter(small_m, -1 - i)
            #assert False 
    input_area_len = tgt_len - guess_len
    mask[:input_area_len, :input_area_len] = torch.triu(torch.full((input_area_len, input_area_len), torch.finfo(dtype).min, dtype=dtype, device=device), diagonal=1)
    mask[input_area_len:, :input_area_len] = torch.zeros((guess_len, input_area_len), dtype=dtype, device=device)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def j_make_causal_mask_multilevel(
    level_sizes: list,is_prefill:bool, WINDOW_SIZE: int, guess : list, guess_size: int, not_seq:bool, continue_all:bool,input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0, extra_input_length = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    Now support input more than 1 token when is_prefill == False
    """
    bsz, tgt_len = input_ids_shape
    tgt_len -= extra_input_length
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    if is_prefill:
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        assert extra_input_length == 0
        assert past_key_values_length == 0
        assert guess is None
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    tiny_mask_size = level_sizes[0] + 1
    mask_cond = torch.arange(tiny_mask_size, device=device)
    hm = mask_cond < (mask_cond + 1).view(tiny_mask_size, 1)

    if guess is not None:
        mask[:,0] = 0
        lguess = len(guess)
        if guess_size == 2:
            small_m = torch.tensor([0, torch.finfo(dtype).min]).repeat(lguess // 2)[:-1]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m, -1)
        elif guess_size == 3:
            small_m1 = torch.tensor([0, 0, torch.finfo(dtype).min]).repeat(lguess // 3)[:-1]
            small_m2 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 3)[:-2]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2)
        elif guess_size == 4:
            small_m1 = torch.tensor([0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 4)[:-1]
            small_m2 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-2]
            small_m3 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-3]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3)
        elif guess_size == 5:
            small_m1 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 5)[:-1]
            small_m2 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-2]
            small_m3 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-3]
            small_m4 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-4]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4)
        elif guess_size == 6:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 6)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-2]
            small_m3 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-3]
            small_m4 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-4]
            small_m5 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-5]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5)
        elif guess_size == 7:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 7)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-2]
            small_m3 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-3]
            small_m4 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-4]
            small_m5 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-5]
            small_m6 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-6]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5).diagonal_scatter(small_m6, -6)

        else:
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0)
            for i in range(guess_size - 1): #7 : 0 - 5
                small_l = [0] * (guess_size - i - 1)  + [torch.finfo(dtype).min] * (i + 1)
                small_m = torch.tensor(small_l).repeat(lguess // guess_size)[:-1 - i]
                mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].diagonal_scatter(small_m, -1 - i)

    for ll in range(len(level_sizes)):
        if ll > 0:
            assert level_sizes[ll] == tiny_mask_size
        mask[tiny_mask_size*ll:tiny_mask_size*(ll+1),:tiny_mask_size].masked_fill_(hm, 0)
        for row in range(1, ll + 1):
            mask[tiny_mask_size*ll:tiny_mask_size*(ll+1),tiny_mask_size*row:tiny_mask_size*(row+1)].fill_diagonal_(0)

    mask = mask.to(dtype)

    if extra_input_length > 0:
        ur = torch.full((extra_input_length, mask.shape[-1]), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.cat([ur, mask], dim=0)
        ul = torch.triu(torch.full((extra_input_length, extra_input_length), torch.finfo(dtype).min, dtype=dtype, device=device), diagonal=1)
        dl = torch.full((tgt_len, extra_input_length), 0, dtype=dtype, device=device)
        ll = torch.cat([ul, dl], dim=0)
        mask = torch.cat([ll, mask], dim=-1)
    
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len + extra_input_length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len + extra_input_length, tgt_len + extra_input_length + past_key_values_length)