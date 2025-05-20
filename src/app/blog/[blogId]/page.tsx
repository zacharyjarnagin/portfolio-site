import { formatStintDate } from "@/app/utils";
import {
  Breadcrumbs,
  Container,
  Link,
  List,
  ListItem,
  Typography,
} from "@mui/material";
import { getAll } from "@vercel/edge-config";

export default async function Page({
  params,
}: {
  params: Promise<{ blogId: string }>;
}) {
  const { blogId } = await params;
  if (!blogId) return <div>404</div>;
  return (
    <div>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Link underline="hover" color="inherit" href="/blog">
          Blogs
        </Link>
        <Typography>{blogId}</Typography>
      </Breadcrumbs>
      {blogId}
    </div>
  );
}
