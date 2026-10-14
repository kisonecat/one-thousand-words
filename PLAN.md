# A Picture Is Worth 1,000 Words

A website that turns a photo into about 1,000 words of ordinary-looking English prose and turns that prose back into the photo. The text reads like a short story about the picture, but every word choice secretly carries part of the image.

## How it works

The system has two parts: a **neural image codec** and **LLM steganography**.

1. **Image → bits.** The photo is cropped square, converted to grayscale and resized to 256×256. A VQGAN encoder compresses it to a 16×16 grid of 256 codebook indices. With a 16,384-entry codebook, each index is 14 bits, so the image becomes a **3,584-bit** stream.
2. **Bits → prose.** A language model writes text in which the image bits, rather than random sampling, choose each next token. This is done with arithmetic coding: the bitstream is treated as if it were compressed text and "decompressed" through the model's next-token probabilities.
3. **Prose → bits → image.** The decoder runs the same model over the same text. At each position it sees which token was chosen and recovers the bits that forced that choice. The VQGAN decoder then rebuilds the photo from the recovered indices.

## Making the text about the photo

Before encoding, a captioning model describes the image, for example "An old man feeding pigeons in a snowy square." This caption is:

- shown as the **visible title** of the document, and
- used as the **prompt** the LLM writes from.

As a result, the hidden pixels live inside a story that is actually about the picture. The decoder reads the title first, so it can rebuild the exact same prompt.

## Capacity budget

| Item | Size |
|---|---|
| Fluent prose capacity | ~3–5 bits/word → ~4,000 bits per 1,000 words |
| Image payload (256 VQGAN codes × 14 bits) | 3,584 bits |
| Header (length, version) | ~32 bits |
| Remaining margin for error correction | ~400 bits |

The main tuning dial is **temperature**. Higher temperature packs more bits into each word, but the prose gets stranger. Lower temperature makes the prose more natural but holds less data.

The VQGAN output keeps the composition, faces and scenes, but fine details are partly hallucinated. Using grayscale means no bits are spent on color.

## Hard problems

- **Determinism.** The encoder and decoder must compute bit-identical probabilities. Floating-point differences across GPUs or browsers will break decoding. The fix is to run one fixed small model on one backend (CPU/WebAssembly) and quantize probabilities to integers before arithmetic coding.
- **Tokenization round-trip.** The generated text must retokenize into exactly the tokens that were generated. During encoding, any token choice that would retokenize ambiguously is excluded.
- **Fragility.** A single edited character, such as autocorrect swapping in smart quotes, corrupts everything after it. To contain this, each paragraph is encoded independently and Reed–Solomon error correction is added over the image bits, so damage stays local.
- **Termination.** The decoder must know where the data ends. A small header encodes the payload length.

## Architecture

Everything runs **in the browser**, so no server or GPU is needed:

- **Image codec:** VQGAN (f16, 16,384 codes), fine-tuned on black-and-white photos and exported to ONNX.
- **Captioner:** a small vision-language model, or simply a caption the user types.
- **Language model:** a small open model (around 0.5B parameters) running on CPU/WebAssembly for determinism.
- **UI:** two modes: *Encode* (upload photo → document) and *Decode* (paste text → photo).

## Build order

1. **Prototype the steganography alone.** Encode random bytes into prose and decode them back, then test the round trip across machines. This is the riskiest part.
2. **Add the image codec.** Measure reconstruction quality at 256 codes on B&W photos.
3. **Add captioning and prompt construction.**
4. **Add error correction and per-paragraph chunking.**
5. **Build the website.**
