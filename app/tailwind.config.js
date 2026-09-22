/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Pearland design tokens
        primary: { DEFAULT: '#1E88E5', dark: '#1565C0', light: '#64B5F6' },
        accent: { DEFAULT: '#43A047', dark: '#2E7D32', light: '#81C784' },
        warning: { DEFAULT: '#FB8C00', dark: '#EF6C00', light: '#FFB74D' },
        danger: { DEFAULT: '#E53935', dark: '#C62828', light: '#EF9A9A' },
        canvas: { DEFAULT: '#F5F5F5', dark: '#14161A' },
        surface: { DEFAULT: '#FFFFFF', dark: '#1D2026' },
        // Chart series steps, validated per mode by the dataviz palette checker.
        series: {
          actual: '#1E88E5',
          predicted: '#E8710A',
          'actual-dk': '#3D92E0',
          'predicted-dk': '#CE7B26',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Roboto', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(16,24,40,0.05), 0 1px 3px rgba(16,24,40,0.08)',
        'card-hover': '0 4px 6px -1px rgba(16,24,40,0.08), 0 2px 4px -2px rgba(16,24,40,0.06)',
      },
      borderRadius: { card: '12px' },
    },
  },
  plugins: [],
}
