// Titles must be printable ASCII (the decoder rebuilds the prompt from them).
export function toTitle(caption) {
  return caption
    .normalize('NFKD')
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[^\x20-\x7e]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\.$/, '');
}
