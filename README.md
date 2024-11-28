# Build-a-Large-Language-Model-From-Scratch

Chapter 1: Build a Large Language Model (From Scratch)

- The “large” in “large language model” refers to both the model’s size in terms of parameters and the immense dataset on which it’s trained.
- Since LLMs are capable of generating text, LLMs are also often referred to as a form of generative artificial intelligence, often abbreviated as generative AI or GenAI.
- Most LLMs today are implemented using the PyTorch deep learning library, which is what we will use.
- Why should we build our own LLMs? Coding an LLM from the ground up is an excellent exercise to understand its mechanics and limitations. Also, it equips us with the required knowledge for pretraining or fine-tuning existing open source LLM architectures to our own domain-specific datasets or tasks.
- The general process of creating an LLM includes pretraining and fine-tuning. The “pre” in “pretraining” refers to the initial phase where a model like an LLM is trained on a large, diverse dataset to develop a broad understanding of language. This pretrained model then serves as a foundational resource that can be further refined through fine-tuning, a process where the model is specifically trained on a narrower dataset that is more specific to particular tasks or domains. This two-stage training approach consisting of pretraining and fine-tuning is depicted in figure below:
  
![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/1.png)

- Readers with a background in machine learning may note that labeling information is typically required for traditional machine learning models and deep neural networks trained via the conventional supervised learning paradigm. However, this is not the case for the pretraining stage of LLMs. In this phase, LLMs use self-supervised learning, where the model generates its own labels from the input data.
- This first training stage of an LLM is also known as pretraining, creating an initial pretrained LLM, often called a base or foundation model. A typical example of such a model is the GPT-3 model (the precursor of the original model offered in ChatGPT). This model is capable of text completion—that is, finishing a half-written sentence provided by a user. It also has limited few-shot capabilities, which means it can learn to perform new tasks based on only a few examples instead of needing extensive training data.
- The two most popular categories of fine-tuning LLMs are **instruction fine-tuning and classification fine-tuning**. In instruction fine-tuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. In classification fine-tuning, the labeled dataset consists of texts and associated class labels—for example, emails associated with “spam” and “not spam” labels.
- Most modern LLMs rely on the transformer architecture, which is a deep neural network architecture introduced in the 2017 paper “Attention Is All You Need” (https://arxiv.org/abs/1706.03762). To understand LLMs, we must understand the original transformer, which was developed for machine translation, translating English texts to German and French. A simplified version of the transformer architecture is depicted below:
![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/2.transformer_architecture.png)


