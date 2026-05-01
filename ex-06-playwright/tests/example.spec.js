import { test, expect } from '@playwright/test';

test('page loads and displays existing images', async ({ page }) => {
  await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

  // Check page title
  await expect(page).toHaveTitle(/TDD Frontend Example/);

  // Check that existing images are displayed
  await expect(page.getByRole('heading', { name: 'AI Alien' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Predator Night Vision' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'ET Bilu' })).toBeVisible();
});

test('can submit a new image', async ({ page }) => {
  await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

  // Fill the form
  await page.getByRole('textbox', { name: 'Image Title' }).fill('Test Image');
  await page.getByRole('textbox', { name: 'Image URL' }).fill('https://via.placeholder.com/150');

  // Submit the form
  await page.getByRole('button', { name: 'Submit' }).click();

  // Check that the new image appears
  await expect(page.getByRole('heading', { name: 'Test Image' })).toBeVisible();
});