# Playwright Tests

This project sets up Playwright to test the vanilla JS web app at https://erickwendel.github.io/vanilla-js-web-app-example/.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Install Playwright browsers (Chrome):
   ```bash
   npx playwright install --with-deps chrome
   ```

## Running Tests

Run the tests locally:
```bash
npm test
```

## Configuration

- **baseURL**: https://erickwendel.github.io/vanilla-js-web-app-example/
- **timeout**: 5 seconds for actions
- **browser**: Chromium only

## CI/CD

GitHub Actions workflow is configured in `.github/workflows/playwright.yml` to run tests on push and pull requests to main/master branches. It installs dependencies, sets up Chromium, runs tests, and uploads HTML reports on failure.