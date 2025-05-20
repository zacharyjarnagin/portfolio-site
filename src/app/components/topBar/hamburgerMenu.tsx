import MenuIcon from "@mui/icons-material/Menu";
import { Grid } from "@mui/material";
import IconButton from "@mui/material/IconButton";

export default function HamburgerMenu({
  toggleDrawer,
}: {
  toggleDrawer: (newOpen: boolean) => () => void;
}) {
  return (
    <Grid size={2} sx={{ display: { xs: "flex", md: "none" } }}>
      <IconButton
        size="large"
        aria-controls="menu-appbar"
        aria-haspopup="true"
        onClick={toggleDrawer(true)}
        color="inherit"
      >
        <MenuIcon />
      </IconButton>
    </Grid>
  );
}
