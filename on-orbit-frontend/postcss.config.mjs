/** @type {import('postcss-load-config').Config} */
// Tailwind 4 moved its PostCSS plugin into a separate package; the bare
// `tailwindcss` plugin entry no longer exists.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
