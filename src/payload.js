// The bitstream hidden in a story: the image codes followed by a CRC-16 so
// the decoder can tell a damaged story from a good one.

import { codesToBits, bitsToCodes, PAYLOAD_BITS as IMAGE_BITS } from './titok.js';

export const CRC_BITS = 16;
export const PAYLOAD_BITS = IMAGE_BITS + CRC_BITS;

// CRC-16/CCITT-FALSE, computed bit by bit over a bit array.
function crc16(bits) {
  let crc = 0xffff;
  for (const b of bits) {
    const top = ((crc >> 15) & 1) ^ b;
    crc = (crc << 1) & 0xffff;
    if (top) crc ^= 0x1021;
  }
  return crc;
}

export function pack(codes) {
  const bits = codesToBits(codes);
  const crc = crc16(bits);
  for (let i = CRC_BITS - 1; i >= 0; i--) bits.push((crc >> i) & 1);
  return bits;
}

// Returns { codes, ok }; ok is false if the CRC does not match.
export function unpack(bits) {
  const image = bits.slice(0, IMAGE_BITS);
  let crc = 0;
  for (let i = 0; i < CRC_BITS; i++) crc = crc * 2 + (bits[IMAGE_BITS + i] ?? 0);
  return { codes: bitsToCodes(image), ok: bits.length >= PAYLOAD_BITS && crc16(image) === crc };
}
