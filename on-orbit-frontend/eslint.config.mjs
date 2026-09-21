// ESLint 9 uses flat config; the old .eslintrc.json is no longer read.
// eslint-config-next 16 ships native flat-config arrays, so these are spread
// directly -- no FlatCompat shim needed. This is the direct equivalent of the
// previous `{"extends": ["next/core-web-vitals", "next/typescript"]}`.
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypeScript from "eslint-config-next/typescript";

const config = [
  { ignores: [".next/**", "node_modules/**", "out/**", "next-env.d.ts"] },
  ...nextCoreWebVitals,
  ...nextTypeScript,
];

export default config;
