// Prompts for the two halves of a story.
//
// The story is only the story: no title travels with it. So the hidden bits
// can't depend on the caption. Instead the caption steers the *opening* (the
// first OPENING_SENTENCES sentences, written by ordinary sampling, carrying no
// bits), and the bits are hidden in a continuation that the decoder can
// re-prompt from the opening alone:
//
//   subjectPrompt(caption)  -> opening sentences      (encoder only)
//   storyPrompt(opening)    -> continuation with bits (encoder and decoder)
//
// The wording was picked by trying variants on SmolLM2-135M: asking it to
// "begin directly" stops it writing an essay about the story instead.

export const OPENING_SENTENCES = 2;

const SYSTEM = '<|im_start|>system\nYou are a gifted novelist.<|im_end|>\n';

export function subjectPromptText(subject) {
  subject = subject.charAt(0).toLowerCase() + subject.slice(1);
  return (
    SYSTEM +
    `<|im_start|>user\nWrite a short story about ${subject}. ` +
    'Begin directly with the story itself.<|im_end|>\n' +
    '<|im_start|>assistant\n'
  );
}

export function storyPromptText(opening) {
  return (
    SYSTEM +
    '<|im_start|>user\nWrite a short story. Begin directly with the story itself.<|im_end|>\n' +
    `<|im_start|>assistant\n${opening}`
  );
}

const ids = (tokenizer, text) => tokenizer.encode(text, { add_special_tokens: false });
export const subjectPromptIds = (tokenizer, subject) => ids(tokenizer, subjectPromptText(subject));
export const storyPromptIds = (tokenizer, opening) => ids(tokenizer, storyPromptText(opening));

// Splits a (normalized) story after its OPENING_SENTENCES-th [.!?]; returns
// null if it has fewer.
export function splitOpening(story) {
  let seen = 0;
  for (let i = 0; i < story.length; i++) {
    if ('.!?'.includes(story[i]) && ++seen === OPENING_SENTENCES) {
      return { opening: story.slice(0, i + 1), rest: story.slice(i + 1) };
    }
  }
  return null;
}
