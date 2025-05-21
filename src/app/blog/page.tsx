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
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "My Blog",
  description: "These are all my blogs",
};

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
                primary={blogData.data.title}
                secondary={blogData.data.date}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );
}
