# Build-a-Large-Language-Model-From-Scratch

1. [Chapter 1: Build a Large Language Model (From Scratch)](#1)
2. [Chapter 6: Fine-tuning for classification](#2)  
   1. [Different categories of fine-tuning](#3)
   2. [Choosing the right approach](#4)
   3. [Data preparation](#5)
   4. [Initializing a model with pre-trained weights](#6)
   5. [Adding a classification head](#7)
   6. [Fine-tuning](#8)
      1. [Fine-tuning selected layers vs. all layers](#9)
      2. [Freezing the model is the first step](#10)
      3. [Calculating the classification loss and accuracy](#11)
      4. [Fine-tuning the model on supervised data](#12)
      5. [Using the LLM as a spam classifier](#13)
   7. [Summary](#14)

<a name="1"></a>
## Chapter 1: Build a Large Language Model (From Scratch)

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
- The transformer architecture consists of two submodules: an encoder and a decoder. The encoder module processes the input text and encodes it into a series of numerical representations or vectors that capture the contextual information of the input. Then, the decoder module takes these encoded vectors and generates the output text. In a translation task, for example, the encoder would encode the text from the source language into vectors, and the decoder would decode these vectors to generate text in the target language. Both the encoder and decoder consist of many layers connected by a so-called self-attention mechanism. You may have many questions regarding how the inputs are preprocessed and encoded.
- A key component of transformers and LLMs is the self-attention mechanism (not shown), which allows the model to weigh the importance of different words or tokens in a sequence relative to each other. This mechanism enables the model to capture long-range dependencies and contextual relationships within the input data, enhancing its ability to generate coherent and contextually relevant output. However, due to its complexity, we will defer further explanation to chapter 3, where we will discuss and implement it step by step.
- Later variants of the transformer architecture, such as BERT (short for bidirectional encoder representations from transformers) and the various GPT models (short for generative pretrained transformers), built on this concept to adapt this architecture for different tasks.
- BERT, which is built upon the original transformer’s encoder submodule, differs in its training approach from GPT. While GPT is designed for generative tasks, BERT and its variants specialize in masked word prediction, where the model predicts masked or hidden words in a given sentence, as shown in figure below. This unique training strategy equips BERT with strengths in text classification tasks, including sentiment prediction and document categorization. As an application of its capabilities, as of this writing, X (formerly Twitter) uses BERT to detect toxic content. GPT, on the other hand, focuses on the decoder portion of the original transformer architecture and is designed for tasks that require generating texts. This includes machine translation, text summarization, fiction writing, writing computer code, and more.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/3.bert_vs_gpt.png)

- up to Transformers vs. LLMs


<a name="2"></a>
## Chapter 6: Fine-tuning for classification

<a name="3"></a>
### Different categories of fine-tuning

  - The most common ways to fine-tune language models are instruction fine-tuning and classification fine-tuning.
    - Instruction fine-tuning involves training a language model on a set of tasks using specific instructions to improve its ability to understand and execute tasks described in natural language prompts.
    - In classification fine-tuning, a concept you might already be acquainted with if you have a background in machine learning, the model is trained to recognize a specific set of class labels, such as “spam” and “not spam.” Examples of classification tasks extend beyond LLMs and email filtering: they include identifying different species of plants from images; categorizing news articles into topics like sports, politics, and technology; and distinguishing between benign and malignant tumors in medical imaging. The key point is that a classification fine-tuned model is restricted to predicting classes it has encountered during its training. For instance, it can determine whether something is “spam” or “not spam,” as illustrated in figure 6.3, but it can’t say anything else about the input text.

In contrast to the classification fine-tuned model depicted in figure 6.3, an instruction fine-tuned model typically can undertake a broader range of tasks. We can view a classification fine-tuned model as highly specialized, and generally, it is easier to develop a specialized model than a generalist model that works well across various tasks.

<a name="4"></a>
### Choosing the right approach
Instruction fine-tuning improves a model’s ability to understand and generate responses based on specific user instructions. Instruction fine-tuning is best suited for models that need to handle a variety of tasks based on complex user instructions, improving flexibility and interaction quality. Classification fine-tuning is ideal for projects requiring precise categorization of data into predefined classes, such as sentiment analysis or spam detection.

While instruction fine-tuning is more versatile, it demands larger datasets and greater computational resources to develop models proficient in various tasks. In contrast, classification fine-tuning requires less data and compute power, but its use is confined to the specific classes on which the model has been trained.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/4.fine_tune_classification_steps.png) 

<a name="5"></a>
### Data preparation

we need to perform:

  - Data balancing
  - Data splitting into train/test/val
  - Creating Pytorch Data Loaders, which require us to perform padding/truncation
  - Padding/truncating the validation and test sets to match the length of the longest training sequence
  - Creating batches of data

<a name="6"></a>
### Initializing a model with pre-trained weights
  - we load a pre-trained GPT model
  - Although his model is good in text completion its performance to perform the classification is very poor:

     text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    
    print(token_ids_to_text(token_ids, tokenizer))

    which returns back:
    
    Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'
    
    The following text 'spam'? Answer with 'yes' or 'no': 'You are a winner
 
    Based on the output, it’s apparent that the model is struggling to follow instructions. This result is expected, as it has only undergone pretraining and lacks instruction fine-tuning. So, let’s prepare the model for classification fine-tuning.

<a name="7"></a>
### Adding a classification head

  - We must modify the pretrained LLM to prepare it for classification fine-tuning. To do so, we replace the original output layer, which maps the hidden representation to a vocabulary of 50,257, with a smaller output layer that maps to two classes: 0 (“not spam”) and 1 (“spam”), as shown in figure 6.9. We use the same model as before, except we replace the output layer.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/5.Adding_a_classification_head.png)

Before we attempt the modification shown in figure above, let’s print the model architecture via print(model):

    GPTModel(
      (tok_emb): Embedding(50257, 768)
      (pos_emb): Embedding(1024, 768)
      (drop_emb): Dropout(p=0.0, inplace=False)
      (trf_blocks): Sequential(
        (0): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (1): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (2): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (3): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (4): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (5): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (6): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (7): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (8): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (9): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (10): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
        (11): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_resid): Dropout(p=0.0, inplace=False)
        )
      )
      (final_norm): LayerNorm()
      (out_head): Linear(in_features=768, out_features=50257, bias=False)
      )

As previously discussed, the GPTModel consists of embedding layers followed by 12 identical transformer blocks (only the last block is shown for brevity), followed by a final LayerNorm and the output layer, out_head.

Next, we replace the out_head with a new output layer (see figure 6.9) that we will fine-tune.

<a name="8"></a>
### Fine-tuning 

<a name="9"></a>
#### Fine-tuning selected layers vs. all layers

Since we start with a pretrained model, it’s not necessary to fine-tune all model layers. In neural network-based language models, the lower layers generally capture basic language structures and semantics applicable across a wide range of tasks and datasets. So, fine-tuning only the last layers (i.e., layers near the output), which are more specific to nuanced linguistic patterns and task-specific features, is often sufficient to adapt the model to new tasks. A nice side effect is that it is computationally more efficient to fine-tune only a small number of layers. 

<a name="10"></a>
#### Freezing the model is the first step 

Technically, training the output layer we just added is sufficient. However, as I found in experiments, fine-tuning additional layers can noticeably improve the predictive performance of the model. (For more details, refer to appendix B.) We also configure the last transformer block and the final LayerNorm module, which connects this block to the output layer, to be trainable, as depicted in figure 6.10.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/6.trainable_layers.png)

Remember that we are interested in fine-tuning this model to return a class label indicating whether a model input is “spam” or “not spam.” We don’t need to fine-tune all four output rows; instead, we can focus on a single output token. In particular, we will focus on the last row corresponding to the last output token, as shown in figure 6.11.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/7.last_output_token.png)

We still need to convert the values into a class-label prediction. But first, let’s understand why we are particularly interested in the last output token only.

We have already explored the attention mechanism, which establishes a relationship between each input token and every other input token, and the concept of a causal attention mask, commonly used in GPT-like models (see chapter 3). This mask restricts a token’s focus to its current position and the those before it, ensuring that each token can only be influenced by itself and the preceding tokens, as illustrated in figure 6.12.

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/8.causal_attention_mechanism.png)

Given the causal attention mask setup in figure 6.12, the last token in a sequence accumulates the most information since it is the only token with access to data from all the previous tokens. Therefore, in our spam classification task, we focus on this last token during the fine-tuning process.

We are now ready to transform the last token into class label predictions and calculate the model’s initial prediction accuracy. Subsequently, we will fine-tune the model for the spam classification task.

<a name="11"></a>
### Calculating the classification loss and accuracy

<a name="12"></a>
### Fine-tuning the model on supervised data

<a name="13"></a>
### Using the LLM as a spam classifier

<a name="14"></a>
### Summary

- There are different strategies for fine-tuning LLMs, including classification fine-tuning and instruction fine-tuning.
- Classification fine-tuning involves replacing the output layer of an LLM via a small classification layer.
- In the case of classifying text messages as “spam” or “not spam,” the new classification layer consists of only two output nodes. Previously, we used the number of output nodes equal to the number of unique tokens in the vocabulary (i.e., 50,256).
- Instead of predicting the next token in the text as in pretraining, classification fine-tuning trains the model to output a correct class label—for example, “spam” or “not spam.”
- The model input for fine-tuning is text converted into token IDs, similar to pretraining.
- Before fine-tuning an LLM, we load the pretrained model as a base model.
- Evaluating a classification model involves calculating the classification accuracy (the fraction or percentage of correct predictions).
- Fine-tuning a classification model uses the same cross entropy loss function as when pretraining the LLM.
