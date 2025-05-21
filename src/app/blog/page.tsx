import {
  Breadcrumbs,
  Link,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
} from "@mui/material";
import { getSortedBlogsData } from "../../../lib/blogs";

export default async function Page() {
  const allBlogsData = getSortedBlogsData();
  return (
    <div>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Typography>Blogs</Typography>
      </Breadcrumbs>
      <Typography variant="h1">My Blog</Typography>
      <List>
        {allBlogsData.map((blogData) => (
          <ListItem key={blogData.slug}>
            <ListItemButton href={`blog/${blogData.slug}`}>
              <ListItemText
                primary={blogData.frontMatter.title}
                secondary={blogData.frontMatter.date}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );
}
