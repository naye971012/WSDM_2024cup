from transformers import T5Tokenizer

model_name = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(model_name,
                                            model_max_length = 1024, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                            legacy = False,
                                              device_map = "balanced",  
                                              max_memory={0: "20GB", 1: "20GB"})


vocab = tokenizer.get_vocab()


for i, value in enumerate(vocab):
    print(f"{i}: {value}")