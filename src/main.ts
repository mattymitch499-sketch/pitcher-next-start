import { mount } from 'svelte';
import App from './App.svelte';
import './styles.css';
import './validation.css';
import './pitcher-details.css';
import './learn-details.css';
import './pitcher-page.css';
import './typography.css';
import './distribution.css';
import './status-nav.css';
import './workload.css';
import './validation-page.css';
import './editorial-theme.css';

mount(App, { target: document.getElementById('app')! });
