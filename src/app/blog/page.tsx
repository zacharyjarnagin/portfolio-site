import { Breadcrumbs, Link, Typography } from "@mui/material";
import { getSortedBlogsData } from "../../../lib/blogs";

export default async function Page() {
  const allPostsData = getSortedBlogsData();
  return (
    <div>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Typography>Blogs</Typography>
      </Breadcrumbs>
      <Typography variant="h1">My Blog</Typography>
      {allPostsData.map(({ id, date, title }) => (
        <li key={id}>
          <a href={`blog/${id}`}>
            {title}
            <br />
            {id}
            <br />
            {date}
          </a>
        </li>
      ))}
    </div>
  );
}
