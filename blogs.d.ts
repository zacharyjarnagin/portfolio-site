type BlogData = {
  title: string;
  date: string;
  summary?: string;
  image?: string;
  bibliographyFilePath?: string;
};

type Blog = {
  slug: string;
  data: BlogData;
  content: string;
};
