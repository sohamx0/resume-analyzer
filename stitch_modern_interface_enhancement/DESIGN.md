---
name: 
colors:
  surface: '#161311'
  surface-dim: '#161311'
  surface-bright: '#3d3836'
  surface-container-lowest: '#110d0c'
  surface-container-low: '#1f1b19'
  surface-container: '#231f1d'
  surface-container-high: '#2e2927'
  surface-container-highest: '#393431'
  on-surface: '#eae1dd'
  on-surface-variant: '#d8c3ad'
  inverse-surface: '#eae1dd'
  inverse-on-surface: '#342f2d'
  outline: '#a08e7a'
  outline-variant: '#534434'
  surface-tint: '#ffb95f'
  primary: '#ffc174'
  on-primary: '#472a00'
  primary-container: '#f59e0b'
  on-primary-container: '#613b00'
  inverse-primary: '#855300'
  secondary: '#ffb783'
  on-secondary: '#4f2500'
  secondary-container: '#d97722'
  on-secondary-container: '#451f00'
  tertiary: '#ffbea0'
  on-tertiary: '#561f00'
  tertiary-container: '#f79a6c'
  on-tertiary-container: '#72300a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffddb8'
  primary-fixed-dim: '#ffb95f'
  on-primary-fixed: '#2a1700'
  on-primary-fixed-variant: '#653e00'
  secondary-fixed: '#ffdcc5'
  secondary-fixed-dim: '#ffb783'
  on-secondary-fixed: '#301400'
  on-secondary-fixed-variant: '#713700'
  tertiary-fixed: '#ffdbcc'
  tertiary-fixed-dim: '#ffb693'
  on-tertiary-fixed: '#351000'
  on-tertiary-fixed-variant: '#76330d'
  background: '#161311'
  on-background: '#eae1dd'
  surface-variant: '#393431'
typography:
  h1:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  h2:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  h3:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: '0'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: '0'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: '0'
  label-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: '1'
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 40px
  container-max: 1280px
  gutter: 24px
---

## Brand & Style
The design system is defined by a sophisticated amber-forward aesthetic, blending the technical precision of high-end software with the organic warmth of laboratory-grown amber. The target audience includes professionals in data science, high-tech engineering, and creative intelligence who value a focused, low-fatigue environment.

The design style is a hybrid of **Minimalism** and **Glassmorphism**. It utilizes deep, warm-toned neutrals to ground the interface while employing translucent, frosted layers to create a sense of depth and computational "breathing room." The emotional response is intended to be one of calm authority—a high-tech sanctuary that feels both cutting-edge and human-centric.

## Colors
This design system employs a "Deep Hearth" palette. The background is a profound charcoal-brown, providing a high-contrast foundation for the amber accents.

- **Primary Amber (#F59E0B):** Used for primary actions, critical data highlights, and active states.
- **Soft Orange (#FB923C):** Used for secondary accents, hover states, and interactive gradients.
- **Deep Resin (Tertiary):** A muted, dark amber used for subtle borders and low-priority backgrounds.
- **Neutrals:** Based on a warm grayscale. Surfaces use a slightly lighter brown-charcoal to distinguish them from the pure dark background.
- **Accents:** Vibrant ambers should be used sparingly as "light sources" within the dark environment.

## Typography
The typography relies exclusively on **Inter** for its utilitarian clarity and modern geometric construction. 

For headlines, use tighter letter spacing and heavier weights to create a "technical" look. For body text, the line height is generous to ensure readability against the dark background. Labels and micro-copy should utilize medium weights and slight tracking (letter spacing) to maintain legibility when rendered in amber or light-gray tones.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a 12-column structure for desktop and a single column for mobile. A 4px baseline grid ensures vertical rhythm.

Spacing is designed to be airy, reflecting the "Intelligence" aspect of the brand. Significant padding within containers prevents the dark theme from feeling cramped. Elements are grouped using generous margins (24px+) to allow the glassmorphic background blurs to be appreciated.

## Elevation & Depth
Depth is created through **Glassmorphism** and tonal layering rather than traditional heavy shadows.

- **Level 0 (Base):** Deepest charcoal background.
- **Level 1 (Cards/Panels):** Semi-transparent surfaces (80% opacity) with a `20px` backdrop-blur.
- **Level 2 (Modals/Popovers):** Higher transparency (60% opacity) with a `40px` backdrop-blur and a subtle `1px` inner border in a faint amber tint to simulate light catching the edge of the glass.
- **Shadows:** Use extremely soft, wide-dispersion shadows with a slight amber hue (`#F59E0B` at 5-10% opacity) to suggest that interactive elements are glowing from within.

## Shapes
The shape language is consistently defined by a **12px (0.75rem)** radius for standard components like cards and input fields. This specific curvature balances the precision of the tech aesthetic with the "warmth" of the amber theme.

- Small elements (buttons, chips) use the 12px base.
- Large containers (sections, main panels) use a 24px (1.5rem) radius.
- Buttons should never be fully pill-shaped; they must maintain the structured 12px corner to feel architectural and professional.

## Components
- **Buttons:** Primary buttons use a solid amber gradient (`#F59E0B` to `#FB923C`) with dark text. Secondary buttons use a "ghost" style: transparent background, 1px amber border, and amber text.
- **Glass Cards:** Use a 1px border with a linear gradient (top-left: white at 10% opacity, bottom-right: amber at 5% opacity) to create a subtle light-refraction effect.
- **Input Fields:** Darker than the surface level, with a 1px deep-resin border. On focus, the border glows with the primary amber and the background blur increases.
- **Chips/Badges:** Small, low-opacity amber backgrounds with high-saturation amber text. Corners follow the 12px rule but scaled down (e.g., 6px).
- **Progress Indicators:** Use glowing amber "pulses" for loading states, emphasizing the "Intelligence" narrative through fluid, light-based animation.
- **Lists:** Separated by low-contrast, warm-tinted dividers. Hover states on list items should use a subtle tint of amber at 5% opacity.