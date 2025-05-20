import { Grid } from "@mui/material";
import Button from "@mui/material/Button";
import { useRouter } from "next/navigation";

export default function DesktopNavList({ pages }: { pages: Pages }) {
  const router = useRouter();

  return (
    <Grid size={2} sx={{ display: "flex" }}>
      {pages.map((page) => (
        <Button
          key={page.label}
          onClick={() => router.push(page.route)}
          color="inherit"
          sx={{ display: { xs: "none", md: "inherit" } }}
        >
          {page.label}
        </Button>
      ))}
    </Grid>
  );
}
