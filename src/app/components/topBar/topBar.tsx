"use client";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Container from "@mui/material/Container";

import { Grid } from "@mui/material";
import SideBarDrawer from "./sideBarDrawer";
import { useState } from "react";
import Title from "./title";
import DesktopNavList from "./desktopNavList";
import HamburgerMenu from "./hamburgerMenu";
import LinkedInIcon from "@mui/icons-material/LinkedIn";

const pages: Pages = [
  { label: "Home", route: "/" },
  { label: "Blog", route: "/blog" },
  {
    label: "LinkedIn",
    route: "https://www.linkedin.com/in/zachary-jarnagin/",
    icon: <LinkedInIcon />,
  },
];

function TopBar() {
  const [open, setOpen] = useState(false);
  const toggleDrawer = (newOpen: boolean) => () => {
    setOpen(newOpen);
  };

  return (
    <AppBar position="sticky" color="default">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Grid
            container
            alignItems="center"
            width="100%"
            justifyContent="center"
            direction="row"
          >
            <HamburgerMenu toggleDrawer={toggleDrawer} />
            <Title />
            <DesktopNavList pages={pages} />
          </Grid>
        </Toolbar>
      </Container>
      <SideBarDrawer pages={pages} open={open} toggleDrawer={toggleDrawer} />
    </AppBar>
  );
}
export default TopBar;
