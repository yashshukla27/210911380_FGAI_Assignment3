"""Small tests protect causal masking, leakage prevention, metrics and token shifting."""
import unittest
import torch
from src.basic import BasicTransformer,PAD,BOS,EOS,collate_basic
from src.common import clean_pair,unique_pairs,scores

class PipelineTests(unittest.TestCase):
    def test_normalization_and_cross_split_duplicates(self):
        a=clean_pair({'en':'  Hello   friend ','hi':' नमस्ते मित्र '})
        self.assertEqual(a['en'],'Hello friend')
        se,sh=set(),set()
        self.assertEqual(len(unique_pairs([a,a],se,sh)),1)
        b=clean_pair({'en':'HELLO FRIEND','hi':'नमस्कार दोस्त'})
        self.assertEqual(unique_pairs([b],se,sh),[])
        c=clean_pair({'en':'Greetings friend','hi':'नमस्ते मित्र'})
        self.assertEqual(unique_pairs([c],se,sh),[])

    def test_causal_mask_prevents_future_token_leakage(self):
        torch.manual_seed(3)
        model=BasicTransformer(30,d_model=16,nhead=2,layers=1,ff_dim=32,dropout=0.).eval()
        src=torch.tensor([[4,5,EOS]])
        a=torch.tensor([[BOS,6,7,8]])
        b=torch.tensor([[BOS,6,20,21]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(src,a)[:,:2],model(src,b)[:,:2],atol=1e-6))

    def test_backward_and_padding(self):
        class Tokenizer:
            def encode(self,text): return [4,5] if len(text)>1 else [4]
        src,tgt=collate_basic([{'en':'ab','hi':'cd'},{'en':'a','hi':'b'}],Tokenizer())
        self.assertEqual(tgt[0,0],BOS)
        self.assertEqual(tgt[1,-1],PAD)
        model=BasicTransformer(20,16,2,1,32,0.)
        logits=model(src,tgt[:,:-1])
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,20),tgt[:,1:].reshape(-1),ignore_index=PAD)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.output.weight.grad.abs().sum().item(),0.)

    def test_metrics_alignment_and_identity(self):
        s=['यह एक बहुत अच्छा उदाहरण है ।','आज मौसम काफी सुहाना और ठंडा है ।']
        m=scores(s,s)
        self.assertAlmostEqual(m['bleu'],100.)
        self.assertAlmostEqual(m['chrf'],100.)
        with self.assertRaises(ValueError): scores(s,s[:1])

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
