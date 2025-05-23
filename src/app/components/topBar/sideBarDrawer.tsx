import { useRouter } from "next/navigation";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import { Box, ListItemIcon } from "@mui/material";

export default function SideBarDrawer({
  pages,
  open,
  toggleDrawer,
}: {
  pages: Pages;
  open: boolean;
  toggleDrawer: (newOpen: boolean) => () => void;
}) {
  const router = useRouter();

  return (
    <Drawer open={open} onClose={toggleDrawer(false)}>
      <Box
        sx={{ width: 250 }}
        role="presentation"
        onClick={toggleDrawer(false)}
      >
        <List>
          {pages.map((page) => (
            <ListItem key={page.label} disablePadding>
              <ListItemButton
                onClick={() =>
                  page.icon ? window.open(page.route) : router.push(page.route)
                }
              >
                {page.icon ? (
                  <ListItemIcon>{page.icon}</ListItemIcon>
                ) : (
                  <ListItemText primary={page.label} />
                )}
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}
