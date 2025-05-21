import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
 
const postsDirectory = path.join(process.cwd(), 'blogs');

const referenceDirectory = "references"
 
export function getSortedBlogsData() {
  // Get file names under /posts
  const fileNames = fs.readdirSync(postsDirectory).filter((fileName) => fileName !== referenceDirectory);
  const allPostsData = fileNames.map((fileName) => {
    // Remove ".md" from file name to get id
    const id = fileName.replace(/\.md$/, '');
 
    // Read markdown file as string
    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
 
    // Use gray-matter to parse the post metadata section
    const matterResult = matter(fileContents);
 
    // Combine the data with the id
    return {
      id,
      ...matterResult.data,
    };
  });
  // Sort posts by date
  return allPostsData.sort((a, b) => {
    if (a.date < b.date) {
      return 1;
    } else {
      return -1;
    }
  });
}

export interface PostFrontMatter {
  title: string;
  date: string;
  tags?: string[];
}

export interface Post {
  slug: string;
  frontMatter: PostFrontMatter;
  content: string;
}


export function getBlogContent(slug: string) {
  const fullPath = path.join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, "utf8");
  const { data, content } = matter(fileContents);

  return {
    slug,
    frontMatter: data,
    content,
  };
}