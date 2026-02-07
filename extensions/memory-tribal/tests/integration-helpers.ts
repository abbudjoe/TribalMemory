/**
 * Helper functions for integration tests to reduce duplication.
 */

/**
 * Find a /v1/recall call in the mock fetch calls.
 */
export function getRecallCall(mockFetch: any): any[] | undefined {
  return mockFetch.mock.calls.find(
    (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/recall"),
  );
}

/**
 * Find a /v1/remember call in the mock fetch calls.
 */
export function getRememberCall(mockFetch: any): any[] | undefined {
  return mockFetch.mock.calls.find(
    (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
  );
}

/**
 * Parse the JSON body from a fetch call.
 */
export function parseCallBody(call: any[]): any {
  return JSON.parse(call[1].body);
}

/**
 * Assert that a result is defined and return it with proper typing.
 * Throws an error if result is null/undefined.
 */
export function assertDefined<T>(result: T | null | undefined, message = "Expected result to be defined"): T {
  if (result === null || result === undefined) {
    throw new Error(message);
  }
  return result;
}
