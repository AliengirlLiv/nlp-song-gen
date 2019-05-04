#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
):

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        text_body = tf.placeholder(tf.int32, [batch_size, None])
        text_prompt = tf.placeholder(tf.int32, [batch_size, None])
        num_gen_steps = tf.placeholder(tf.int32, [1])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        logits = sample.PERPLEXITY2(
            hparams=hparams, length=num_gen_steps,
            text_body=text_body,
            text_prompt=text_prompt,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
            )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
       
        filename = "../openai_dset/test_preprocessed.txt"
        with open(filename, "r") as f:
            all_samples = f.read().split("<|endoftext|>\n")
            print("NUM SAMPLES", len(all_samples))
        count = 0
        total_perplexity = 0
        for s in all_samples:
            sentences = s.split("\n")
            first_sentence = sentences.pop(0).split()
            first = first_sentence.pop(0)
            first_sentence = " ".join(first_sentence)
            sentences = "\n".join([first_sentence] + sentences) + "<|endoftext|>"
            print("FIRST LINE")
            print(first)
            print("REMAINDER")
            print(sentences)

            prompt_tokens = enc.encode(first)
            context_tokens = enc.encode(sentences) ### THIS IS THE TARGETS
            targets = tf.convert_to_tensor(prompt_tokens[1:] + context_tokens, dtype=tf.int32)
            targets = tf.expand_dims(targets, axis=0)
            weights = [0 for _ in range(len(prompt_tokens) - 1)] + [1 for _ in range(len(context_tokens))] # THIS IS THE WEIGHTS
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
            weights = tf.expand_dims(weights, axis=0)
            generated = 0
            for _ in range(nsamples // batch_size):
                loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
                perplexity = tf.exp(loss)
                out = sess.run(perplexity, feed_dict={
                    text_body: [context_tokens for _ in range(batch_size)],
                    text_prompt: [prompt_tokens for _ in range(batch_size)],
                    num_gen_steps: [len(context_tokens)]
                    })
                
                for i in range(batch_size):
                    generated += 1
        
                    print("   Perplexity: " + str(out))
                    total_perplexity = (out + total_perplexity * count)/(count + 1)
                    count += 1
                    print("Total: " + str(total_perplexity))

if __name__ == '__main__':
    fire.Fire(interact_model)


