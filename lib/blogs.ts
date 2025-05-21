import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
 
const blogsDirectory = path.join(process.cwd(), 'blogs');

const referenceDirectory = "references"

export function getBlogContent(slug: string): Blog {
  const fullPath = path.join(blogsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, "utf8");
  const { data, content } = matter(fileContents);

  return {
    slug,
    frontMatter: data,
    content,
  };
}
 
export function getSortedBlogsData(): Blog[] {
  // Get file names under /blogs
  const fileNames = fs.readdirSync(blogsDirectory).filter((fileName) => fileName !== referenceDirectory);
  const allBlogsData: Blog[] = fileNames.map((fileName) => {
    // Remove ".md" from file name to get id
    const id = fileName.replace(/\.md$/, '');
    return getBlogContent(id);
  });
  // Sort blogs by date
  return allBlogsData.sort((a, b) => {
    if (a.frontMatter.date < b.frontMatter.date) {
      return 1;
    } else {
      return -1;
    }
  });
}