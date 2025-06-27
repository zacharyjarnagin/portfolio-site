---
title: 'How this site was made'
date: '2025-06-26'
summary: 'The hows and whys of this site'
---

## NextJS

I used [Next.js](https://nextjs.org) as it is now the [recommended way of creating a new React app](https://react.dev/learn/creating-a-react-app), not to mention I have used it in the past and thoroughly enjoyed it. The [app router](https://nextjs.org/docs/app) is intuitive and easy to use, and I have found site performance to be excellent out of the box.

## Vercel

[Vercel](https://vercel.com/) is used to [deploy](https://vercel.com/docs/deployments) the app, [host images](https://vercel.com/docs/vercel-blob), and [store a list](https://vercel.com/docs/edge-config) of my experiences in JSON. Getting [started with Vercel](https://vercel.com/docs/getting-started-with-vercel) was very easy and having deployments triggered on merges to main in my GitHub repo is a great win for no additional setup.

## Typescript

Is there anything better?

## MaterialUI

[MaterialUI](https://mui.com/material-ui/integrations/nextjs/) was used simply because it looks good and has a great [ecosystem of components](https://mui.com/material-ui/all-components/). I didn't want to spend a lot of time on design but I found MUI gave me enough control over the component styles when I needed it.

## Rehype/Remark

To create my blog, I wanted to keep things in markdown for easy editing and creation of new blogs. I wanted to simply create a new file, merge to main, and have my new blog entry appear. And I achieved that!

I use [react-markdown](https://github.com/remarkjs/react-markdown) to render the markdown and metadata is loaded from [gray-matter](https://www.npmjs.com/package/gray-matter). An assortment of [rehype](https://github.com/rehypejs) and [remark](https://github.com/remarkjs/remark) plugins.
