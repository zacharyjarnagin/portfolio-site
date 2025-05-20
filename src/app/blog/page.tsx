import { Breadcrumbs, Link, Typography } from "@mui/material";

export default async function Page() {
  return (
    <div>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Typography>Blogs</Typography>
      </Breadcrumbs>
      <Typography variant="h1">My Blog</Typography>
    </div>
  );
}
