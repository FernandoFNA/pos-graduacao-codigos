import { test, expect } from '@playwright/test';

test.describe('Form Submission and Validation', () => {
  test('should submit form and update the image list', async ({ page }) => {
    await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

    // Count initial articles
    const initialCount = await page.locator('article').count();

    // Fill form
    await page.getByPlaceholder('Image Title').fill('New Test Image');
    await page.getByPlaceholder('https://img.com/erick.png').fill('https://via.placeholder.com/150');

    // Submit
    await page.getByRole('button', { name: 'Submit Form' }).click();

    // Check list updated
    await expect(page.locator('article')).toHaveCount(initialCount + 1);
    await expect(page.getByRole('heading', { name: 'New Test Image' })).toBeVisible();

    // Check form cleared
    await expect(page.getByPlaceholder('Image Title')).toHaveValue('');
    await expect(page.getByPlaceholder('https://img.com/erick.png')).toHaveValue('');
  });

  test('should show validation error for empty title', async ({ page }) => {
    await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

    // Leave title empty, fill URL
    await page.getByPlaceholder('Image Title').fill('');
    await page.getByPlaceholder('https://img.com/erick.png').fill('https://via.placeholder.com/150');

    // Submit
    await page.getByRole('button', { name: 'Submit Form' }).click();

    // Check error message
    await expect(page.getByText('Please type a title for the image.')).toBeVisible();

    // List should not update
    const count = await page.locator('article').count();
    expect(count).toBe(3); // Assuming 3 initial
  });

  test('should show validation error for empty URL', async ({ page }) => {
    await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

    // Fill title, leave URL empty
    await page.getByPlaceholder('Image Title').fill('Test Title');
    await page.getByPlaceholder('https://img.com/erick.png').fill('');

    // Submit
    await page.getByRole('button', { name: 'Submit Form' }).click();

    // Check error message
    await expect(page.getByText('Please type a valid URL')).toBeVisible();

    // List should not update
    const count = await page.locator('article').count();
    expect(count).toBe(3);
  });

  test('should show validation error for invalid URL', async ({ page }) => {
    await page.goto('https://erickwendel.github.io/vanilla-js-web-app-example/');

    // Fill title, invalid URL
    await page.getByPlaceholder('Image Title').fill('Test Title');
    await page.getByPlaceholder('https://img.com/erick.png').fill('invalid-url');

    // Submit
    await page.getByRole('button', { name: 'Submit Form' }).click();

    // Check error message
    await expect(page.getByText('Please type a valid URL')).toBeVisible();

    // List should not update
    const count = await page.locator('article').count();
    expect(count).toBe(3);
  });
});