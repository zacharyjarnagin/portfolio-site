import {
  Box,
  Breadcrumbs,
  Card,
  CardActionArea,
  CardContent,
  CardMedia,
  Link,
  Typography,
} from "@mui/material";
import { getSortedBlogsData } from "../../../lib/blogs";
import type { Metadata } from "next";
import { formatDateStr } from "../utils";

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
      <Box sx={{ my: 5, gap: 2, display: "grid", justifyItems: "center" }}>
        {allBlogsData.map((blogData) => (
          <Card sx={{ width: "80%", borderRadius: 2 }} key={blogData.slug}>
            <CardActionArea href={`blog/${blogData.slug}`}>
              <CardContent>
                {blogData.data.image && (
                  <CardMedia
                    component="img"
                    height="256"
                    sx={{ borderRadius: 2 }}
                    src={blogData.data.image}
                    alt="green iguana"
                  />
                )}
                <Typography variant="h3">{blogData.data.title}</Typography>
                {blogData.data.summary && (
                  <Typography variant="subtitle1">
                    {blogData.data.summary}
                  </Typography>
                )}
                <Typography variant="overline">
                  {formatDateStr(blogData.data.date, true)}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        ))}
      </Box>
    </div>
  );
}
