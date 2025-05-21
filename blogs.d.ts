type BlogData = {
  title: string;
  date: string;
  bibliographyFilePath?: string;
}

type  Blog = {
  slug: string;
  data: BlogData;
  content: string;
}