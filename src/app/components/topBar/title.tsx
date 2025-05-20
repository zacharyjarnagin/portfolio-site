import { Grid, Avatar, Typography } from "@mui/material";
export default function Title() {
  return (
    <>
      <Grid
        size={2}
        justifyContent="center"
        sx={{ display: { xs: "none", md: "flex" } }}
      >
        <Avatar
          alt="Zachary Jarnagin"
          src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/zach-professional-2xX2dxKBLSsB5oEdDVgZvDt0eTEs6J.jpeg"
        />
      </Grid>
      <Grid size={"grow"} textAlign={"center"} justifyContent="center">
        <Typography
          variant="h1"
          noWrap
          component="a"
          href="/"
          sx={{
            fontFamily: "monospace",
            fontWeight: 700,
            letterSpacing: ".3rem",
            color: "inherit",
            textDecoration: "none",
            fontSize: { xs: 16, sm: 18, md: 24 },
          }}
        >
          ZACHARY JARNAGIN
        </Typography>
      </Grid>
    </>
  );
}
