import { Grid, IconButton } from "@mui/material";
import Button from "@mui/material/Button";
import { useRouter } from "next/navigation";

export default function DesktopNavList({ pages }: { pages: Pages }) {
  const router = useRouter();

  return (
    <Grid size={2} sx={{ display: "flex" }}>
      {pages.map((page) =>
        page.icon ? (
          <IconButton
            key={page.label}
            onClick={() => window.open(page.route)}
            sx={{ display: { xs: "none", md: "inherit" } }}
          >
            {page.icon}
          </IconButton>
        ) : (
          <Button
            key={page.label}
            onClick={() => router.push(page.route)}
            color="inherit"
            sx={{ display: { xs: "none", md: "inherit" } }}
            startIcon={page.icon}
          >
            {!page.icon && page.label}
          </Button>
        )
      )}
    </Grid>
  );
}
