// The shareable document is just the story.
//
// The encoder only ever emits printable ASCII with single spaces (see
// vocab.js), so on the way back in we can safely undo what email clients,
// word processors and line wrapping tend to do to text.

export function normalizeStory(text) {
  return text
    .replace(/[‘’‚‛′]/g, "'")
    .replace(/[“”„‟″]/g, '"')
    .replace(/…/g, '...')
    .replace(/[  -   　]/g, ' ')
    .replace(/[​-‍﻿]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
