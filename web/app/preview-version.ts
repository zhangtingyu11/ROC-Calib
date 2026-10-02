/** Resolve a real preview version, including saved results after annotation edits.
 * This is display state only; it does not mark a stale calibration as current.
 * Callers supply versions already scoped to the active sensors and camera model.
 */
export function resolvePreviewVersion<T extends { id: string; createdAt: string; source: string }>(
  versions: readonly T[], requestedId: string, preferredId: string,
): T | null {
  const explicit = versions.find(version => version.id === requestedId);
  if (explicit) return explicit;
  const preferred = versions.find(version => version.id === preferredId);
  if (preferred) return preferred;
  const calculated = versions.filter(version => version.source === "calculated");
  return [...(calculated.length ? calculated : versions)]
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0] ?? null;
}
