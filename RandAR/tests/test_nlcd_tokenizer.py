import numpy as np

from RandAR.model.nlcd_tokenizer import NLCDTokenizer

def test_decode_codes_to_img():
    tokenizer = NLCDTokenizer(vocab_size=25)
    codes = np.arange(16).reshape(1, 16)  # batch_size=1, seq_len=16
    decode_table = np.ones((25, 2, 2)) * 11  # All pixels map to NLCD class 11
    result = tokenizer.decode_codes_to_img(codes, decode_table, scale=10)
    
    # Check shape: 16 patches of 2x2 in 4x4 grid = 8x8 image, then upscaled 10x = 80x80
    assert result.shape == (1, 80, 80, 3)
    assert result.dtype == np.uint8
    

if __name__ == "__main__":
    test_decode_codes_to_img()