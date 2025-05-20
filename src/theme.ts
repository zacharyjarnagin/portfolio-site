'use client';
import { createTheme } from '@mui/material/styles';
import type {} from '@mui/lab/themeAugmentation';

const theme = createTheme({
  colorSchemes: {
    dark: true,
  },
  typography: {
    fontFamily: 'var(--font-roboto)',
  },
  cssVariables: true,
});

export default theme;
