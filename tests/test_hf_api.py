"""Local tiny-config API checks; these are not pretrained model evaluations."""
import unittest
import torch
from transformers import MT5Config,MT5ForConditionalGeneration,Adafactor
from transformers import Gemma3TextConfig,Gemma3ForCausalLM,LlamaConfig,LlamaForCausalLM

class HuggingFaceAPITests(unittest.TestCase):
    def test_mt5_finetuning_step(self):
        m=MT5ForConditionalGeneration(MT5Config(vocab_size=32,d_model=16,d_kv=8,d_ff=32,
            num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
        m.gradient_checkpointing_enable();m.config.use_cache=False
        loss=m(input_ids=torch.tensor([[2,3,1]]),labels=torch.tensor([[4,5,1]])).loss
        loss.backward()
        Adafactor(m.parameters(),lr=.0005,relative_step=False,scale_parameter=False).step()
        self.assertTrue(torch.isfinite(loss))

    def test_llama_and_gemma_generation_api(self):
        models=[LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,bos_token_id=1,eos_token_id=2,pad_token_id=0)),
            Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,bos_token_id=1,
            eos_token_id=2,pad_token_id=0,sliding_window=16))]
        for model in models:
            out=model.eval().generate(input_ids=torch.tensor([[1,4,5]]),max_new_tokens=2,do_sample=False,pad_token_id=0)
            self.assertGreater(out.shape[1],3)
            self.assertLessEqual(out.shape[1],5)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
