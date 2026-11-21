we just put out inkling-2-mini-unbound, an uncensored version of inkling 2 mini

https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound

if you've tried the other uncensored inkling 2 mini versions you've probably noticed they stop refusing, but a lot of the answers come out kind of vague. we dug into why. inkling has a small conv layer right after attention, and it quietly mixes back in most of what a normal abliteration takes out. so people push the edit way harder to compensate, and the answers get worse

we built the edit around that conv, so it can be a lot lighter. only 20 attention weight matrices changed, experts untouched, and you can drop it in anywhere you run the original

on a test set we kept held back the whole time:

- refused 0 of 277 harmful prompts (the original refuses pretty much all of them)
- 54-63% of answers are actually useful, about 50% more than openuncensored's version (https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) on the same prompts and judge
- stopped refusing harmless stuff like "how do i kill a python process"
- same scores as the original on mmlu-pro, gsm8k, math, gpqa, ifeval and humaneval+

one tip: if you run it at high thinking effort, set a thinking budget. it can overthink on edgy prompts

eval code and the exact edit are up if you want to check our work: https://github.com/nokeepai/inkling-2-mini-unbound

it'll do pretty much whatever you ask, so be sensible with it. happy to answer questions
