import {
  createStyles,
  Menu,
  Center,
  Header,
  Container,
  Group,
  Image,
} from "@mantine/core";
import { IconChevronDown } from "@tabler/icons";
import { useNavigate, NavLink } from "react-router-dom";
import { UnstyledButton } from "@mantine/core";

import logo from "../images/logo.png";

const HEADER_HEIGHT = 80;

const useStyles = createStyles((theme) => ({
  inner: {
    height: HEADER_HEIGHT,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  links: {
    [theme.fn.smallerThan("sm")]: {
      display: "none",
    },
  },

  burger: {
    [theme.fn.largerThan("sm")]: {
      display: "none",
    },
  },

  link: {
    display: "block",
    lineHeight: 1,
    padding: "8px 12px",
    borderRadius: theme.radius.sm,
    textDecoration: "none",
    color:
      theme.colorScheme === "dark"
        ? theme.colors.dark[0]
        : theme.colors.gray[7],
    fontSize: theme.fontSizes.sm,
    fontWeight: 500,

    "&:hover": {
      backgroundColor:
        theme.colorScheme === "dark"
          ? theme.colors.dark[6]
          : theme.colors.gray[0],
    },
  },

  linkLabel: {
    marginRight: 5,
  },
}));

export default function HeaderAction({ links }) {
  const { classes } = useStyles();

  const navigate = useNavigate();
  const items = links.map((link) => {
    const menuItems = link.links?.map((item) => (
      <Menu.Item key={item.label} onClick={() => navigate(item.link)}>
        {item.label}
      </Menu.Item>
    ));

    if (menuItems) {
      return (
        <Menu key={link.label} trigger="hover" exitTransitionDuration={0}>
          <Menu.Target>
            <NavLink to={link.link} className={classes.link}>
              <Center>
                <span className={classes.linkLabel}>{link.label}</span>
                <IconChevronDown size={12} stroke={1.5} />
              </Center>
            </NavLink>
          </Menu.Target>
          <Menu.Dropdown>{menuItems}</Menu.Dropdown>
        </Menu>
      );
    }

    return (
      <NavLink className={classes.link} to={link.link}>
        {link.label}
      </NavLink>
    );
  });

  return (
    <Header height={HEADER_HEIGHT} sx={{ borderBottom: 0, position: "sticky" }}>
      <Container className={classes.inner} fluid>
        <UnstyledButton>
          <Image
            src={logo}
            alt="ML Toolkit Logo"
            width={160}
            ml={5}
            onClick={() => {
              navigate("/");
            }}
          />
        </UnstyledButton>
        <Group spacing={5} className={classes.links}>
          {items}
        </Group>
        <a href="https://ko-fi.com/W7W4I0P3R" target="_blank">
          <img
            height="36"
            style={{ border: "0px", height: "36px" }}
            src="https://storage.ko-fi.com/cdn/kofi2.png?v=3"
            border="0"
            alt="Buy Me a Coffee at ko-fi.com"
          />
        </a>
      </Container>
    </Header>
  );
}
