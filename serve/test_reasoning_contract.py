"""The endpoint accepts common switches and clamps unsupported efforts."""
import unittest
from serve.frontend import anthropic_to_messages,openai_to_messages,effort_kwargs
from serve.server import Service

class ReasoningContract(unittest.TestCase):
    def test_off_spellings_override_shared_medium_on_openai(self):
        svc=Service.__new__(Service);svc.shared={'reasoning_effort':'medium'}
        variants=[{'reasoning_effort':'none'},{'reasoning':{'effort':'none','enabled':False}},
                  {'thinking':{'type':'disabled'}},{'enable_thinking':False},
                  {'reasoning':{'enabled':False}},{'chat_template_kwargs':{'enable_thinking':False}}]
        for v in variants:
            with self.subTest(v=v):
                req=svc.with_shared({'messages':[{'role':'user','content':'x'}],**v},'openai')
                self.assertEqual(openai_to_messages(req)[2],{'enable_thinking':False})
    def test_off_spellings_override_shared_medium_on_anthropic(self):
        svc=Service.__new__(Service);svc.shared={'reasoning_effort':'medium'}
        for v in [{'reasoning_effort':'none'},{'reasoning':{'enabled':False}},
                  {'thinking':{'type':'disabled'}},{'enable_thinking':False}]:
            req=svc.with_shared({'messages':[{'role':'user','content':'x'}],**v},'anthropic')
            self.assertEqual(anthropic_to_messages(req)[2],{'enable_thinking':False})
    def test_canonical_levels_keep_existing_kwargs(self):
        for k,v in [('none',{'enable_thinking':False}),('low',{'reasoning_effort':'low'}),
                    ('medium',{'reasoning_effort':'medium'}),('high',{'reasoning_effort':'xhigh'}),
                    ('xhigh',{'reasoning_effort':'xhigh'})]:self.assertEqual(effort_kwargs(k),v)
    def test_unsupported_strengths_clamp(self):
        for k,level in [('ultra','xhigh'),('very_high','xhigh'),('very_low','low'),('unrecognised','medium'),(9,'xhigh')]:
            self.assertEqual(effort_kwargs(k),{'reasoning_effort':level})
        self.assertEqual(effort_kwargs(-1),{'enable_thinking':False})

if __name__=='__main__':unittest.main()
