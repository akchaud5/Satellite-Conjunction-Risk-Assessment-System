import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Narrow an unknown caught value to a message string.
 *
 * Replaces the `catch (err: any)` blocks that tripped
 * @typescript-eslint/no-explicit-any and failed `next build`.
 */
export function errorMessage(err: unknown, fallback = "An unknown error occurred."): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === "string" && err) return err;
  return fallback;
}
