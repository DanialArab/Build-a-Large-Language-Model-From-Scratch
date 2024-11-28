# Build-a-Large-Language-Model-From-Scratch

Chapter 1: Build a Large Language Model (From Scratch)

- The “large” in “large language model” refers to both the model’s size in terms of parameters and the immense dataset on which it’s trained.
- Since LLMs are capable of generating text, LLMs are also often referred to as a form of generative artificial intelligence, often abbreviated as generative AI or GenAI.
- Most LLMs today are implemented using the PyTorch deep learning library, which is what we will use.
- Why should we build our own LLMs? Coding an LLM from the ground up is an excellent exercise to understand its mechanics and limitations. Also, it equips us with the required knowledge for pretraining or fine-tuning existing open source LLM architectures to our own domain-specific datasets or tasks.
- The general process of creating an LLM includes pretraining and fine-tuning. The “pre” in “pretraining” refers to the initial phase where a model like an LLM is trained on a large, diverse dataset to develop a broad understanding of language. This pretrained model then serves as a foundational resource that can be further refined through fine-tuning, a process where the model is specifically trained on a narrower dataset that is more specific to particular tasks or domains. This two-stage training approach consisting of pretraining and fine-tuning is depicted in figure below:
  
![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/1.png)

- 
