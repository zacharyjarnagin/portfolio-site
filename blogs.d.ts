type BlogFrontMatter = {
  title: string;
  date: string;
}

type  Blog = {
  slug: string;
  frontMatter: BlogFrontMatter;
  content: string;
}