# Project Documentation

Welcome to the documentation for xplainable. This guide will help you set up and contribute to our Docusaurus-powered documentation, which is maintained exclusively on the `docs` branch.

## Prerequisites

Before you start, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (version 12 or greater)
- [Yarn](https://yarnpkg.com/) or [npm](https://www.npmjs.com/)

## Cloning and Setup

Since our documentation lives on the `docs` branch, you'll need to clone it specifically. Here's how:

```bash
git clone --single-branch --branch docs [Your Repository URL]
cd xplainable
```

Once cloned, install the necessary dependencies:

```bash
yarn install
# Or, if you're using npm
npm install
```

## Running the Documentation Locally
To view and edit the documentation on your local machine, run:

```bash
npx docusaurus start
```

This command starts a local development server and opens up a browser window. 
Most changes will reflect live without needing to restart the server.

## Making Changes
Feel free to make changes to the documentation by editing existing or adding new markdown files in the docs directory. Docusaurus uses Markdown, allowing easy content writing and formatting.

When adding new documentation files, remember to update the sidebar configuration to include them. This is usually managed in `sidebars.js` or via markdown file front matter.

## Building and Testing

To build the site and preview the final version, run:

```bash
npx docusaurus build
```

This command generates static content in the build directory, which you can serve using any static content hosting service for testing.

Support
For questions or support, reach out to us at contact@xplainable.io.

Thank you for contributing to the xplainable documentation!