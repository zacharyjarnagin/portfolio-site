import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
 
const blogsDirectory = path.join(process.cwd(), 'blogs');

const referenceDirectory = "references"

export function getBlogContent(slug: string): Blog {
  const fullPath = path.join(blogsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, "utf8");

  // TypeScript and gray-matter do not appear to play well together. This warrants
  // more investigation
  const { data, content } = matter(fileContents) as unknown as Blog

  return {
    slug,
    data,
    content,
  };
}
 
export function getSortedBlogsData(): Blog[] {
  // Get file names under /blogs, ignoring the directory for bibliographies
  const fileNames = fs.readdirSync(blogsDirectory).filter((fileName) => fileName !== referenceDirectory);
  const allBlogsData: Blog[] = fileNames.map((fileName) => {
    // Remove ".md" from the file name and use it as the slug
    const slug = fileName.replace(/\.md$/, '');
    return getBlogContent(slug);
  });
  // Sort blogs by date
  return allBlogsData.sort((a, b) => {
    if (a.data.date < b.data.date) {
      return 1;
    } else {
      return -1;
    }
  });
}