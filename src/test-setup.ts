import { cleanup } from '@testing-library/svelte';
import { afterEach, vi } from 'vitest';

Object.defineProperty(window, 'scrollTo', { value: vi.fn(), writable: true });

afterEach(() => cleanup());
