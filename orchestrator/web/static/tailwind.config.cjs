module.exports = {
  darkMode: 'class',
  content: [
    './orchestrator/web/static/*.html',
    './orchestrator/web/static/*.js',
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          950: 'rgb(15 17 23 / <alpha-value>)',
          900: 'rgb(23 27 38 / <alpha-value>)',
          800: 'rgb(36 42 59 / <alpha-value>)',
          700: 'rgb(47 53 72 / <alpha-value>)',
        },
      },
    },
  },
  plugins: [],
};
