import { expect, test } from "@playwright/test";

test("operations workspace exposes review and recovery workflows", async ({ page }) => {
  await page.goto("/#/operations");
  await expect(page.getByRole("heading", { name: "Review, recover and repair from one place." })).toBeVisible();
  await expect(page.getByRole("button", { name: /Review queue/ })).toBeVisible();
  await page.getByRole("button", { name: "Eligibility profile" }).click();
  await expect(page.getByText("Eligibility and application defaults")).toBeVisible();
  await page.getByRole("button", { name: "Document health" }).click();
  await expect(page.getByText("Validate every attached PDF before using it.")).toBeVisible();
  await page.getByRole("button", { name: "Recovery", exact: true }).click();
  await expect(page.getByText("Restoring first creates a safety copy of the current state.")).toBeVisible();
});

test("decision centre remains safe by default", async ({ page }) => {
  await page.goto("/#/decisions");
  await expect(page.getByRole("heading", { name: "Know why every job is shown or hidden." })).toBeVisible();
  await expect(page.getByRole("button", { name: "Create backup" })).toBeVisible();
  await expect(page.getByText("Actionable", { exact: true }).first()).toBeVisible();
});

test("applications and document preview remain reachable", async ({ page }) => {
  await page.goto("/#/applications");
  await expect(page.getByRole("heading", { name: "Your search, clearly organised." })).toBeVisible();
  await page.goto("/#/preview");
  await expect(page.getByText("Résumé library", { exact: true })).toBeVisible();
});
