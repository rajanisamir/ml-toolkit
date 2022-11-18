export function getWhitespaceList(list) {
  const maxLength = Math.max(...list.map((str) => str.length));
  return list.map((str) => "\u00A0".repeat(maxLength - str.length));
}
