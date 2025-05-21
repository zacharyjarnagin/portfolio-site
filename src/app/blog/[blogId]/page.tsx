import { Breadcrumbs, Container, Link, Typography } from "@mui/material";
import { getBlogContent } from "../../../../lib/blogs";
import remarkGfm from "remark-gfm";
import { MarkdownAsync } from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";

export default async function Page({
  params,
}: {
  params: Promise<{ blogId: string }>;
}) {
  const { blogId } = await params;
  const { frontMatter, content } = await getBlogContent(blogId);
  if (!blogId) return <div>404</div>;
  return (
    <div>
      {/* light theme for syntax blocks */}
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/highlight.js@11.8.0/styles/github.min.css"
        media="(prefers-color-scheme: light)"
      />
      {/* dark theme for syntax blocks */}
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/highlight.js@11.8.0/styles/github-dark.min.css"
        media="(prefers-color-scheme: dark)"
      />
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
      />
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Link underline="hover" color="inherit" href="/blog">
          Blogs
        </Link>
        <Typography>{frontMatter.title}</Typography>
      </Breadcrumbs>
      <Container maxWidth="md" sx={{ py: 4, overflowX: "hidden" }}>
        <Typography variant="h2" component="h1" gutterBottom>
          {frontMatter.title}
        </Typography>
        <Typography variant="subtitle2" color="text.secondary">
          {new Date(frontMatter.date).toLocaleDateString()}
        </Typography>
        <MarkdownAsync
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[
            [
              rehypeCitation,
              {
                bibliography: frontMatter.bibliography,
                path: process.cwd(),
                csl: "apa",
                lang: "en-US",
              },
            ],
            rehypeKatex,
            rehypeHighlight,
            rehypeRaw,
          ]}
        >
          {content}
        </MarkdownAsync>
      </Container>
    </div>
  );
}
